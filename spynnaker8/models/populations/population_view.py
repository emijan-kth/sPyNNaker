# Copyright (c) 2017-2019 The University of Manchester
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import logging
from spynnaker.pyNN.models.populations import PopulationView as _BaseClass
logger = logging.getLogger(__name__)


class PopulationView(_BaseClass):
    """ A view of a subset of neurons within a\
    :py:class:`~spynnaker.pyNN.models.populations.Population`.

    In most ways, Populations and PopulationViews have the same behaviour,
    i.e., they can be recorded, connected with Projections, etc.
    It should be noted that any changes to neurons in a PopulationView
    will be reflected in the parent Population and *vice versa.*

    It is possible to have views of views.

    .. note::
        Selector to Id is actually handled by :py:class:`AbstractSized`.

    .. deprecated:: 6.0
        Use
        :py:class:`spynnaker.pyNN.models.populations.PopulationView` instead.
    """
    __slots__ = []

    def __init__(self, parent, selector, label=None):
        """
        :param parent: the population or view to make the view from
        :type parent: ~spynnaker.pyNN.models.populations.Population or
            ~spynnaker.pyNN.models.populations.PopulationView
        :param selector: a slice or numpy mask array.
            The mask array should either be a boolean array (ideally) of the
            same size as the parent,
            or an integer array containing cell indices,
            i.e. if `p.size == 5` then:

            ::

                PopulationView(p, array([False, False, True, False, True]))
                PopulationView(p, array([2, 4]))
                PopulationView(p, slice(2, 5, 2))

            will all create the same view.
        :type selector: None or slice or int or list(bool) or list(int) or
            ~numpy.ndarray(bool) or ~numpy.ndarray(int)
        :param str label: A label for the view
        """
        super(PopulationView, self).__init__(parent, selector, label)
        logger.warning(
            "please use spynnaker.pyNN.models.populations."
            "PopulationView instead")
